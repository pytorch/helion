"""Registry of benchmarkable comparison workloads.

Each submodule registers one or more :class:`Workload` entries via :func:`register`;
the submodules are auto-imported here so adding a kernel is just adding a file (no
edits to this file, so parallel wiring never conflicts). Torch/Helion imports stay
inside each workload's ``build`` callable so importing the registry pulls no CUDA.
"""

from __future__ import annotations

import dataclasses
import importlib
import pkgutil
from collections.abc import Callable

_BuildResult = tuple[Callable[..., object], Callable[..., object], tuple[object, ...]]


@dataclasses.dataclass(frozen=True)
class Workload:
    """One benchmarkable kernel plus its reference and correctness tolerance."""

    kernel_name: str
    rtol: float
    atol: float
    build: Callable[[], _BuildResult]


WORKLOADS: dict[str, Workload] = {}

DEFAULT_WORKLOAD = "rms_norm-2048x1024"


def register(workload_id: str, workload: Workload) -> None:
    """Register a workload under a stable id (last registration wins)."""
    WORKLOADS[workload_id] = workload


for _module in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{_module.name}")
