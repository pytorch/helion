"""Settings of Helion's Inductor backend, set through torch.compile's options
with a "helion." prefix. helion.inductor registers this module with Inductor,
which makes it part of the FX graph cache key."""

# install_config_module() rejects the `annotations` global that
# `from __future__ import annotations` defines.
# ruff: noqa: I002

import hashlib
import pathlib
import sys
import typing
from typing import Literal

from torch.utils._config_module import install_config_module

from ..runtime.settings import _env_get_literal
from ..runtime.settings import _get_backend

# Lower matmuls natively (dot reductions) and emit them as Helion. A matmul that
# HelionChoices declines when lowering (dynamic shapes, dims of size 1, the
# strict and deterministic modes, cpp_wrapper) keeps its extern kernel. One whose
# kernel Helion cannot emit, which only scheduling shows (a CPU scalar in its
# epilogue), fails the compile with Unsupported.
native_matmul: bool = True
# The generated kernels' @helion.kernel(backend=...): "triton" or "cute", the
# backends this one supports. HELION_BACKEND=cute sets "cute", read as
# @helion.kernel reads it (an unknown name raises); every other backend,
# tileir included, leaves "triton".
backend: Literal["triton", "cute"] = "cute" if _get_backend() == "cute" else "triton"
# "none", "quick" or "full"; HELION_AUTOTUNE_EFFORT sets the default, read as
# @helion.kernel reads it: HELION_AUTOTUNE_EFFORT="" leaves "quick", where a
# raw read would generate autotune_effort="" and fail at the kernel's first
# call with KeyError: ''.
autotune_effort: Literal["none", "quick", "full"] = _env_get_literal(
    "HELION_AUTOTUNE_EFFORT",
    "quick",
    mapping={key: key for key in ("none", "quick", "full")},
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

# Not TYPE_CHECKING, which would be a setting.
if typing.TYPE_CHECKING:
    from torch.utils._config_typing import *  # noqa: F403

install_config_module(sys.modules[__name__])
