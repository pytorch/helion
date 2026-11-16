"""General-purpose work-item loop of a warp role (the flash kernels' scaffold).

Every fused flash body gives each warp role one ``if warp_idx ...:`` guard,
a ``head`` that runs once per role (its ``setmaxnreg``, the raw-mbarrier
phases that carry across work items) and a per-item body.  On the flat
one-item grid the body runs once (``flat_role_source``); under a scheduler it
runs inside a loop over the CTA's work items: the strided ``while`` over
tile ids (block / cluster id plus grid / cluster stride), the counted
``cutlass.range`` form, the CLC scheduler's valid-tile loop, the host item
table of the persistent flex walk, the segment tiling's work table.

:class:`PersistentItemLoop` is that loop's shape and :meth:`render` the one
place that lays it out; the scaffolds in ``cute_flash`` (the ws body's
``_flash_persistent_wrap``, the fa4 body's ``_flash_fa4_wrap`` and its
early-head and segment variants) differ only in the lines they supply.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class PersistentItemLoop:
    """One role's work-item loop.

    ``guard`` is the role's ``if warp_idx ...:`` line; ``head`` the lines that
    run once per role, written at the guard body's 4-space indent (``None``
    emits no head line at all, ``""`` an empty one -- the ws and fa4 bodies
    differ there and the generated text is pinned); ``setup`` the lines
    before the loop (unindented; the loop variable's init, the stride, the
    carried decode); ``loop`` the loop statement itself; ``prelude`` the
    per-item block that precedes the body (8-space indented and
    newline-terminated when non-empty: the item decode, the gmem re-slices);
    ``body`` the per-item role code (8-space indented); ``advance`` the
    per-item lines after the body (unindented: the per-item barrier, the
    stride, the carried decode's update); ``tail`` the text after the loop,
    verbatim (each scaffold formats its own tail lines).
    """

    guard: str
    head: str | None
    loop: str
    setup: tuple[str, ...] = ()
    prelude: str = ""
    body: str = ""
    advance: tuple[str, ...] = ()
    tail: str = ""

    def render(self) -> str:
        lines = ["", self.guard]
        if self.head is not None:
            lines.append(self.head)
        lines.extend(f"    {line}" for line in self.setup)
        lines.append(f"    {self.loop}")
        text = "\n".join(lines) + "\n" + self.prelude + self.body
        text += "".join(f"\n        {line}" for line in self.advance)
        return text + self.tail


def flat_role_source(guard: str, head: str, body: str, tail: str = "") -> str:
    """The flat (one work item per CTA) form of a role: ``guard``, then
    ``head`` (verbatim; the scaffold formats it, newline-terminated when it
    is emitted at all), ``body`` and ``tail``."""
    return f"\n{guard}\n{head}{body}{tail}"
