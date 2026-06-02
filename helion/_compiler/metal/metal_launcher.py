from __future__ import annotations

from typing import Any
from typing import cast


def set_required_threads_per_threadgroup(
    metal_kernel: object,
    group_size: tuple[int, int, int],
) -> None:
    cast("Any", metal_kernel).required_threads_per_threadgroup = group_size
