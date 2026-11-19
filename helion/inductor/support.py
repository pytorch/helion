"""What Helion kernels support, which helion.inductor asks both while lowering
(HelionChoices) and while generating code (HelionScheduling)."""

from __future__ import annotations

import functools

import torch
from torch._inductor import config
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

MATMUL_DTYPES = OrderedSet([torch.float16, torch.bfloat16, torch.float32])


def python_wrapper() -> bool:
    return not (V.graph.cpp_wrapper or V.graph.fx_wrapper)


# Inductor's numerics modes, which Helion kernels do not implement yet.
NUMERICS_MODES = (
    "strict_reduction",
    "batch_invariant",
    "deterministic",
    "eager_numerics.division_rounding",
    "eager_numerics.disable_ftz",
    "eager_numerics.use_pytorch_libdevice",
    "emulate_precision_casts",
)


def numerics_modes() -> list[str]:
    """The NUMERICS_MODES that are on."""
    return [
        mode
        for mode in NUMERICS_MODES
        if functools.reduce(getattr, mode.split("."), config)
    ]
