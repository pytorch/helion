"""Bind grouped kernels the way a search that proposes the bounded D shift does."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING
from unittest.mock import patch

from helion._compiler.cute import grouped_d_shift

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def d_shift_proposals() -> Iterator[None]:
    """Lift ``grouped_d_shift.d_shift_searched`` for the kernels bound inside.

    The compiler's shifted seeds, its M,N worklist tiles as it sizes them
    for the shift's fixed descriptors and the paired prefix recipes all
    store through the bounded D shift, so the autotuner proposes none of
    them until that gate goes (it seeds the tiles without the shift
    meanwhile).  Their tests bind under this context: the proposals stay
    covered for the day it does.
    """
    with patch.object(grouped_d_shift, "D_SHIFT_SEARCHED", True):
        yield
