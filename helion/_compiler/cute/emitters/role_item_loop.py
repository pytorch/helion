"""General-purpose work-item loop of a warp role, as AST (the tcgen05 scaffold).

The AST sibling of :mod:`.persistent_item_loop`, which lays the same shape
out as text for the flash scaffolds.  The tcgen05 persistent GEMM body
gives each warp role (TMA load, MMA, epilogue, C-input, scheduler) one
``if <role predicate>:`` block holding a prelude that runs once per role
(the PDL wait, the role's scheduler or mailbox handshake, its counters, the
hoisted role prelude), a ``while`` over the CTA's work items, and a tail
after the loop (the sentinel release, the producer drain).  Those blocks are
spliced into the ``DeviceFunction`` body ahead of the device-AST passes, so
they must stay AST: dependency extraction clones them, the TMA-role hoist
re-anchors them and the mailbox-field pruning walks them.

:class:`RoleItemLoop` is that block's shared shape and
:meth:`RoleItemLoop.build` assembles it from the statements a family
builder in ``tcgen05/scheduler_pids.py`` fills in.  Not every loop of the
scaffold is an item loop: the group-search scheduler warp wraps its whole
loop in dense / consumer-local guards, the full/edge split's first pass
runs ahead of the scheduler reset inside the role prelude, and the
specialized grouped builder nests one loop per quota rung, so those still
build their inner ``while`` by hand and use :class:`RoleItemLoop` for the
role block around them.
:class:`StaticSchedulerHandle` names a role's own CUTLASS persistent tile
scheduler (the static and the grouped group-search schedulers share the
``initial_work_tile_info`` / ``advance_to_next_work`` / ``get_current_work``
protocol and the ``is_valid_tile`` loop test).
"""

from __future__ import annotations

import ast
import dataclasses

from ...ast_extension import create
from ...ast_extension import expr_from_string
from ...ast_extension import statement_from_string


@dataclasses.dataclass
class RoleItemLoop:
    """One role's work-item loop.

    ``predicate`` is the role's guard expression; ``prelude`` the statements
    that run once per role before the loop; ``loop_test`` the ``while``
    condition (``None`` is the one-shot form: the item runs once, inline,
    without ``item_tail``); ``item_head`` the per-item statements that
    precede the role's work (the work-tile decode, the mailbox release);
    ``body`` the role's per-item work (the dependency statements, the role
    statements, the tile-counter bump); ``item_tail`` the per-item
    statements after the work (the scheduler advance, the next mailbox
    wait); ``tail`` the statements after the loop.
    """

    predicate: str
    prelude: list[ast.stmt] = dataclasses.field(default_factory=list)
    loop_test: str | None = None
    item_head: list[ast.stmt] = dataclasses.field(default_factory=list)
    body: list[ast.stmt] = dataclasses.field(default_factory=list)
    item_tail: list[ast.stmt] = dataclasses.field(default_factory=list)
    tail: list[ast.stmt] = dataclasses.field(default_factory=list)

    def build(self) -> ast.If:
        stmts = [*self.prelude]
        item = [*self.item_head, *self.body]
        if self.loop_test is None:
            stmts.extend(item)
        else:
            stmts.append(
                create(
                    ast.While,
                    test=expr_from_string(self.loop_test),
                    body=[*item, *self.item_tail],
                    orelse=[],
                )
            )
        stmts.extend(self.tail)
        return create(
            ast.If, test=expr_from_string(self.predicate), body=stmts, orelse=[]
        )


@dataclasses.dataclass(frozen=True)
class StaticSchedulerHandle:
    """A role's own persistent tile scheduler and its current work tile."""

    sched_var: str
    work_tile_var: str

    def create_stmts(self, create_expr: str) -> list[ast.stmt]:
        """Bind the scheduler to ``create_expr`` and peek at its first tile."""
        return [
            statement_from_string(f"{self.sched_var} = {create_expr}"),
            statement_from_string(
                f"{self.work_tile_var} = {self.sched_var}.initial_work_tile_info()"
            ),
        ]

    def create_static_stmts(self, params_var: str) -> list[ast.stmt]:
        """``StaticPersistentTileScheduler`` over the launch grid from the
        ``PersistentTileSchedulerParams`` bound to ``params_var``."""
        return self.create_stmts(
            "cutlass.utils.StaticPersistentTileScheduler.create("
            f"{params_var}, cute.arch.block_idx(), cute.arch.grid_dim())"
        )

    def advance_stmts(self) -> list[ast.stmt]:
        return [
            statement_from_string(f"{self.sched_var}.advance_to_next_work()"),
            statement_from_string(
                f"{self.work_tile_var} = {self.sched_var}.get_current_work()"
            ),
        ]

    @property
    def loop_test(self) -> str:
        return f"{self.work_tile_var}.is_valid_tile"


def persistent_tile_scheduler_params_stmt(
    params_var: str, params_args: str
) -> ast.stmt:
    return statement_from_string(
        f"{params_var} = cutlass.utils.PersistentTileSchedulerParams({params_args})"
    )
