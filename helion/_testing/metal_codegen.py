"""Helpers for Metal codegen tests, which bind on CPU."""

from __future__ import annotations

import re


def launch_dims(code: str) -> tuple[int, ...]:
    """The launch's threadgroup shape, from the host code's ``_block_dims``."""
    dims = re.search(r"_block_dims=\((\d+), (\d+), (\d+)\)", code)
    assert dims is not None, code
    return tuple(int(d) for d in dims.groups())
