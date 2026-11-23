"""CPU buffers and a runner for hand-built CuTe lane programs.

The tests of the lane-loop passes write a lane loop as Python source, run it
through the passes and execute what comes out as one thread: a tensor is a
``Buffer``, whose ``iterator + offset`` is a ``Pointer`` that loads and stores
one element and counts both.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

from .._compiler.ast_read_writes import ast_rename
from .._compiler.cute.lane_loops.markers import restore_unprocessed_lane_reduce_markers
from .._compiler.cute.lane_loops.split import split_lane_loop_reductions

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np


class Buffer:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values.copy()
        self.loads = 0
        self.stores = 0

    @property
    def iterator(self) -> Pointer:
        return Pointer(self, 0)


class Pointer:
    def __init__(self, buffer: Buffer, offset: int) -> None:
        self.buffer = buffer
        self.offset = offset

    def __add__(self, offset: int) -> Pointer:
        return Pointer(self.buffer, self.offset + offset)

    def load(self) -> float:
        self.buffer.loads += 1
        return float(self.buffer.values[self.offset])

    def store(self, value: float) -> None:
        self.buffer.stores += 1
        self.buffer.values[self.offset] = value


def run_lane_program(
    body: list[ast.AST],
    arrays: Mapping[str, np.ndarray],
    namespace: Mapping[str, object],
    *,
    uniform_names: set[str],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    renames: dict[str, str] | None = None,
    filename: str,
) -> dict[str, Buffer]:
    """Lower ``body``'s lane reductions (the lane split, then the markers it
    left), apply the pending ``renames`` and run the result as one thread on
    a ``Buffer`` per array, with ``namespace`` bound beside the buffers."""
    body = split_lane_loop_reductions(
        body,
        uniform_names=uniform_names,
        proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
        rename_groups=renames,
    )
    body = restore_unprocessed_lane_reduce_markers(body)
    if renames is not None:
        for statement in body:
            ast_rename(statement, renames)
    buffers = {name: Buffer(values) for name, values in arrays.items()}
    module = ast.Module(body=cast("list[ast.stmt]", body), type_ignores=[])
    exec(compile(ast.unparse(module), filename, "exec"), {**buffers, **namespace})
    return buffers
