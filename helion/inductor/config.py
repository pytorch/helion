"""Settings of Helion's Inductor backend. enable() registers this module with
Inductor, which makes it part of the FX graph cache key."""

# install_config_module() rejects the `annotations` global that
# `from __future__ import annotations` defines.
# ruff: noqa: I002

import hashlib
import pathlib
import sys
from typing import TYPE_CHECKING
from typing import Literal

from torch.utils._config_module import Config
from torch.utils._config_module import install_config_module

# Lower matmuls natively (dot reductions) and emit them as Helion. A matmul that
# Helion declines when lowering (dynamic shapes, the strict and deterministic
# modes, cpp_wrapper) keeps its extern kernel or template. Otherwise it is
# already a native matmul, so Triton's native matmul codegen emits it, instead
# of extern_kernels.mm, where it is not a Helion kernel: where Helion declines
# it when scheduling, e.g. with a gathered operand as in a[i] @ b, and where
# Inductor fuses it with a row reduction, as in softmax(a @ b, -1) with N <= 128,
# which Inductor's own codegen path emits. Such cases shrink as this backend
# supports more.
native_matmul: bool = True
# The generated kernels' @helion.kernel(backend=..., autotune_effort=...).
backend: Literal["triton", "cute"] = "triton"
# "none", "quick" or "full"; HELION_AUTOTUNE_EFFORT sets the default, as for
# @helion.kernel.
autotune_effort: str = Config(
    default="quick", env_name_default="HELION_AUTOTUNE_EFFORT"
)
# Helion's fast_math: approximate fp32 division, exp, erf and rsqrt. Off, as in
# Triton kernels, leaves Helion's default, which HELION_FAST_MATH=1 turns on.
fast_math: bool = False


def _hash_sources() -> str:
    package = pathlib.Path(__file__).parent
    files = sorted(package.glob("*.py"))
    return hashlib.sha256(b"".join(path.read_bytes() for path in files)).hexdigest()


# The cache key includes private entries too: a graph whose output code another
# version of helion.inductor generated is not reused.
_source_hash: str = _hash_sources()

if TYPE_CHECKING:
    from torch.utils._config_typing import *  # noqa: F403

install_config_module(sys.modules[__name__])
