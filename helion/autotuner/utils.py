from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import collections.abc


@contextlib.contextmanager
def safe_globals(*classes: type) -> collections.abc.Generator[None, None, None]:
    """Temporarily add classes to torch's safe globals for weights_only loading."""
    prev = torch.serialization.get_safe_globals()
    torch.serialization.add_safe_globals(list(classes))
    try:
        yield
    finally:
        torch.serialization.clear_safe_globals()
        torch.serialization.add_safe_globals(prev)
