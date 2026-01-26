from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from .. import exc
from .ast_read_writes import ReadWrites

if TYPE_CHECKING:
    import ast


class LoopDependencyChecker:
    """
    A class to check dependencies between top-level for loops in a Helion kernel.

    This class tracks memory accesses (reads and writes) for each top-level for loop
    and raises an error if a later loop reads or writes to anything written in a
    previous loop.
    """

    def __init__(self) -> None:
        self.reads: set[str] = set()
        self.writes: set[str] = set()
        self._barrier_after_root: set[int] = set()
        self._root_counter: int = 0
        self.disabled: bool = False

    def insert_barrier_after_root(self, root_id: int) -> None:
        """Record that a barrier separates root_id and root_id+1."""
        self._barrier_after_root.add(root_id)

    def register_loop(self, loop_node: ast.For, root_id: int | None = None) -> None:
        if self.disabled:
            return
        current_root = root_id if root_id is not None else self._root_counter
        if (current_root - 1) in self._barrier_after_root:
            self.reads.clear()
            self.writes.clear()
            self._barrier_after_root.discard(current_root - 1)
        rw = ReadWrites.from_list(loop_node.body)

        self._check_dependencies(rw)

        self.reads |= set(rw.reads)
        self.writes |= set(rw.writes)
        self._root_counter = current_root + 1

    def _check_dependencies(self, rw: ReadWrites) -> None:
        """
        Check for dependencies between the current loop and previous loops.

        Raises:
            exc.LoopDependencyError: If a dependency is detected
        """
        for name in sorted(itertools.chain(rw.reads, rw.writes)):
            if name in self.writes:
                raise exc.LoopDependencyError(name)
